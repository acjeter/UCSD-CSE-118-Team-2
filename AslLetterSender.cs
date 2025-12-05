using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class AslLetterSender : MonoBehaviour
{
    [Header("Server Settings")]
    [Tooltip("Laptop IP address running server.py")]

    // TODO:
    public string serverIp = "192.168.0.219";   // <-- set this to the Mac's IP
    // ------------------------------------------------------------------------

    [Tooltip("TCP port matching tcp_server in server.py")]
    public int serverPort = 5005;

    private TcpClient client;
    private NetworkStream stream;

    void Start()
    {
        Connect();
    }

    void OnDestroy()
    {
        CloseConnection();
    }

    private void Connect()
    {
        try
        {
            Debug.Log($"[AslLetterSender] Connecting to {serverIp}:{serverPort} ...");
            client = new TcpClient();
            client.Connect(serverIp, serverPort);
            stream = client.GetStream();
            Debug.Log("[AslLetterSender] Connected.");
        }
        catch (Exception e)
        {
            Debug.LogError("[AslLetterSender] Failed to connect: " + e.Message);
        }
    }

    private void CloseConnection()
    {
        try
        {
            if (stream != null)
            {
                stream.Close();
                stream = null;
            }
            if (client != null)
            {
                client.Close();
                client = null;
            }
        }
        catch (Exception e)
        {
            Debug.LogError("[AslLetterSender] Error closing connection: " + e.Message);
        }
    }

    private void SendRawJson(string jsonLine)
    {
        if (client == null || !client.Connected)
        {
            Debug.LogWarning("[AslLetterSender] Not connected, attempting reconnect...");
            Connect();
            if (client == null || !client.Connected)
            {
                Debug.LogError("[AslLetterSender] Still not connected. Dropping message.");
                return;
            }
        }

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(jsonLine + "\n"); // newline-delimited
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError("[AslLetterSender] Error sending data: " + e.Message);
            CloseConnection();  // force reconnect next time
        }
    }

    /// <summary>
    /// Call this when your ASL model recognizes a letter.
    /// </summary>
    public void OnNewLetterRecognized(string letter)
    {
        if (string.IsNullOrEmpty(letter))
            return;

        // Normalize to a single upper-case character
        char c = char.ToUpper(letter[0]);
        string json = "{\"letter\":\"" + c + "\"}";

        Debug.Log("[AslLetterSender] Sending: " + json);
        SendRawJson(json);
    }
}